# Description:
# Image correction and other pre-processing steps.

package(default_visibility = ["//moonlight:__subpackages__"])

licenses(["notice"])  # Apache 2.0

py_library(
    name = "preprocessing",
    deps = [":rotation"],
)

py_library(
    name = "rotation",
    srcs = ["rotation.py"],
    deps = [
        "//moonlight/staves:staffline_distance",
        # tensorflow dep
    ],
)

py_test(
    name = "rotation_test",
    srcs = ["rotation_test.py"],
    deps = [
        ":rotation",
        "//moonlight:image",
        # tensorflow dep
    ],
    data = ["//moonlight/testdata:images"],
)
