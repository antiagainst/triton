from __future__ import annotations

import argparse
import hashlib
import io
import json
import pytest
import tarfile
import tempfile

from pathlib import Path

import triton

from triton.runtime.build import compile_module_from_src

TEST_MODULE_C = """
#include <Python.h>
#include <string.h>

static PyObject* go(PyObject* self, PyObject* args) {
    const char *command;
    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;

    const char* res;
    if (strcmp(command, "hello") == 0) {
        res = "hiya";
    } else {
        res = "huh";
    }
    return PyUnicode_FromString(res);
}

static PyMethodDef ModuleMethods[] = {
  {"go", go, METH_VARARGS, "test_module.go for testing"},
  {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ModuleDef = {
  PyModuleDef_HEAD_INIT,
  "test_module",
  NULL, //documentation
  -1, //size
  ModuleMethods
};

PyMODINIT_FUNC PyInit_test_module(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if(m == NULL) {
    return NULL;
  }
  PyModule_AddFunctions(m, ModuleMethods);
  return m;
}
"""


def test_compile_module(fresh_triton_cache):
    mod = compile_module_from_src(TEST_MODULE_C, "test_module")

    with pytest.raises(Exception):
        mod.go()

    assert mod.go("huh") == "huh"
    assert mod.go("hello") == "hiya"

    # Make sure the module is cached
    mod2 = compile_module_from_src(TEST_MODULE_C, "test_module")
    assert mod2.__file__ == mod.__file__


def test_compile_module_bad_cache(fresh_knobs):
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        called_get_file = False

        class InvalidFileCacheManager(triton.runtime.cache.FileCacheManager):

            def get_file(self, filename: str) -> str | None:
                nonlocal called_get_file
                called_get_file = True
                (tmp / filename).write_text("not an so")
                return str(tmp / filename)

        # First corrupt the cache
        fresh_knobs.cache.manager_class = InvalidFileCacheManager

        mod = compile_module_from_src(TEST_MODULE_C, "test_module")
        assert called_get_file

        with pytest.raises(Exception):
            mod.go()

        assert mod.go("huh") == "huh"
        assert mod.go("hello") == "hiya"


def _amd_build_helpers(monkeypatch, base_dir):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]))
    import build_helpers

    monkeypatch.setattr(build_helpers, "get_base_dir", lambda: str(base_dir))
    for source, contents in [
        ("third_party/amd/backend/codegen/CMakeLists.txt", "project(triton_amd_codegen)\n"),
        ("third_party/amd/backend/codegen/amdgpu_codegen.cc", "int codegen;\n"),
        ("include/triton/Tools/LLVMOptions.h", "#pragma once\n"),
        ("lib/Tools/LLVMOptions.cpp", "int options;\n"),
    ]:
        path = base_dir / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    return build_helpers


def _amd_build_helper_args(build_helpers, tmp_path):
    parser = argparse.ArgumentParser()
    build_helpers.add_common_args(parser)
    return build_helpers.normalize_parsed_args(
        parser.parse_args([
            "--triton-cache-path",
            str(tmp_path / "cache"),
            "--triton-llvm-system-suffix",
            "ubuntu-arm64",
            "--llvm-syspath",
            str(tmp_path / "unrelated-core-llvm"),
        ]))


@pytest.mark.parametrize("artifact", ["sdk", "codegen"])
@pytest.mark.parametrize(
    "mirror", ["", "https://mirror.example/generic/oaiartifacts", "https://mirror.example/generic/oaiartifacts/"])
def test_amd_codegen_download_mirror(artifact, mirror, monkeypatch, tmp_path):
    build_helpers = _amd_build_helpers(monkeypatch, tmp_path)
    monkeypatch.setenv("OAIARTIFACTS_BASE_URL", mirror)
    monkeypatch.delenv("TRITON_CACHE_DEPENDENCY_DOWNLOADS", raising=False)
    args = _amd_build_helper_args(build_helpers, tmp_path)

    llvm_hash = "123456789abcdef"
    llvm_info = {"llvm_hash": llvm_hash, "sdk": {"build_number": 1, "sha256sum": {}}}
    revision = build_helpers.get_amd_codegen_revision(llvm_info)
    if artifact == "sdk":
        name = f"amd-llvm-{llvm_hash[:8]}-ubuntu-arm64-1"
        relative_path = "lib/cmake/llvm/LLVMConfig.cmake"
    else:
        name = f"amd-codegen-{llvm_hash[:8]}-ubuntu-arm64-{revision}"
        relative_path = f"lib/{build_helpers.amd_codegen_library_name()}"
    contents = b"test artifact"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo(f"{name}/{relative_path}")
        member.size = len(contents)
        archive.addfile(member, io.BytesIO(contents))
    archive_bytes = buffer.getvalue()
    checksum = hashlib.sha256(archive_bytes).hexdigest()
    if artifact == "sdk":
        llvm_info["sdk"]["sha256sum"]["ubuntu-arm64"] = checksum
    else:
        llvm_info["codegen"] = {"revision": revision, "sha256sum": {"ubuntu-arm64": checksum}}

    downloads = []

    def download(url, path, label):
        downloads.append(url)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(archive_bytes)

    monkeypatch.setattr(build_helpers, "_download_file", download)
    if artifact == "sdk":
        result = Path(build_helpers.download_amd_llvm_sdk(llvm_info, args)) / relative_path
    else:
        (tmp_path / "cmake").mkdir()
        (tmp_path / "cmake" / "amd-llvm-info.json").write_text(json.dumps(llvm_info))
        build_helpers.download_and_copy_amd_codegen(args)
        result = tmp_path / "third_party" / "amd" / "backend" / relative_path

    base = "https://mirror.example/generic/triton-llvm" if mirror else (
        "https://oaitriton.blob.core.windows.net/public/llvm-builds")
    assert downloads == [f"{base}/{name}.tar.gz"]
    assert result.read_bytes() == contents


def test_amd_codegen_revision_tracks_inputs(monkeypatch, tmp_path):
    build_helpers = _amd_build_helpers(monkeypatch, tmp_path)
    llvm_info = {"llvm_hash": "123456789abcdef", "sdk": {"build_number": 1}}
    revision = build_helpers.get_amd_codegen_revision(llvm_info)

    # Files that are not compiled, and Windows line endings, leave it unchanged.
    codegen_dir = tmp_path / "third_party" / "amd" / "backend" / "codegen"
    (codegen_dir / "amdgpu_codegen.cc.orig").write_text("stale backup\n")
    source = codegen_dir / "amdgpu_codegen.cc"
    source.write_bytes(source.read_bytes().replace(b"\n", b"\r\n"))
    assert build_helpers.get_amd_codegen_revision(llvm_info) == revision

    # New or edited sources, and a different SDK, each change it.
    revisions = {revision}
    (codegen_dir / "Helper.h").write_text("#pragma once\n")
    revisions.add(build_helpers.get_amd_codegen_revision(llvm_info))
    (tmp_path / "lib" / "Tools" / "LLVMOptions.cpp").write_text("int options = 1;\n")
    revisions.add(build_helpers.get_amd_codegen_revision(llvm_info))
    revisions.add(build_helpers.get_amd_codegen_revision({**llvm_info, "sdk": {"build_number": 2}}))
    assert len(revisions) == 4


def test_amd_codegen_builds_unpublished_revision(monkeypatch, tmp_path):
    build_helpers = _amd_build_helpers(monkeypatch, tmp_path)
    args = _amd_build_helper_args(build_helpers, tmp_path)
    checksum = {"ubuntu-arm64": "0" * 64}
    llvm_info = {
        "llvm_hash": "123456789abcdef",
        "sdk": {"build_number": 1, "sha256sum": checksum},
        # Published from other sources, so it must not be downloaded.
        "codegen": {"revision": "000000000000", "sha256sum": checksum},
    }
    (tmp_path / "cmake").mkdir()
    (tmp_path / "cmake" / "amd-llvm-info.json").write_text(json.dumps(llvm_info))
    built = tmp_path / "build" / build_helpers.amd_codegen_library_name()
    built.parent.mkdir()
    built.write_bytes(b"source build")

    def build(amd_llvm_info, helper_args, install_path=None):
        assert amd_llvm_info == llvm_info and install_path is None
        return str(built)

    def download(url, path, label):
        pytest.fail(f"downloaded {url}")

    monkeypatch.setattr(build_helpers, "build_amd_codegen", build)
    monkeypatch.setattr(build_helpers, "_download_file", download)
    build_helpers.download_and_copy_amd_codegen(args)
    assert (tmp_path / "third_party" / "amd" / "backend" / "lib" / built.name).read_bytes() == b"source build"
