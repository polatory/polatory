if(VCPKG_TARGET_IS_WINDOWS)
  vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY ONLY_DYNAMIC_CRT)
else()
  set(VCPKG_BUILD_TYPE release)
  set(VCPKG_POLICY_MISMATCHED_NUMBER_OF_BINARIES enabled)
  vcpkg_check_linkage(ONLY_STATIC_LIBRARY)
endif()

string(REGEX MATCH "^[0-9]+\\.[0-9]+" short_version "${VERSION}")

if(VCPKG_TARGET_IS_LINUX)
  set(filename "arm-performance-libraries_${short_version}_deb_gcc.tar")
  set(sha512 43f76d1fe629dfcd679f64442c6aacf7a04dc471263631af3b1d339c1207d87b7d2fd6314c3dc940fa6cdefd68310551ead8456d2eb783f06175177aeb3a864c)
elseif(VCPKG_TARGET_IS_OSX)
  set(filename "arm-performance-libraries_${short_version}_macOS.tgz")
  set(sha512 15270a1d1a95f5f87fe94549ddeafb246c5957746bcd082482ea4efc7cf6490cd45f80ea3269131e2dcfa2276f2aac6d3c0e349898e4ac41f222917df7e12610)
elseif(VCPKG_TARGET_IS_WINDOWS)
  set(filename "arm-performance-libraries_${short_version}_Windows.msi")
  set(sha512 da3d5cdc65a0d5f686e8e1a731f11fc5253aba8db4bd747deeebd4e5063840f06b2953d1fdc0371ac568d645cd68df31be5f6da6f3e7039bd8680a3ce544ca2b)
endif()

vcpkg_download_distfile(ARCHIVE
  URLS "https://developer.arm.com/-/cdn-downloads/permalink/Arm-Performance-Libraries/Version_${short_version}/${filename}"
  FILENAME "${filename}"
  # Arm's CDN rejects vcpkg's User-Agent.
  HEADERS "User-Agent: curl"
  SHA512 ${sha512}
)

set(archive_dir "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-archive")
file(REMOVE_RECURSE "${archive_dir}")
vcpkg_extract_archive(ARCHIVE "${ARCHIVE}" DESTINATION "${archive_dir}")

if(VCPKG_TARGET_IS_LINUX)
  file(GLOB installer "${archive_dir}/*/*_deb.sh")
  set(package_dir "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-packages")
  file(REMOVE_RECURSE "${package_dir}")
  file(MAKE_DIRECTORY "${package_dir}")
  # --accept accepts Arm's EULA.
  vcpkg_execute_required_process(
    COMMAND bash "${installer}" --accept --save-packages-to "${package_dir}"
    WORKING_DIRECTORY "${CURRENT_BUILDTREES_DIR}"
    LOGNAME "save-packages-${TARGET_TRIPLET}"
  )

  set(deb_dir "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-deb")
  file(REMOVE_RECURSE "${deb_dir}")
  vcpkg_extract_archive(ARCHIVE "${package_dir}/armpl_${short_version}_gcc.deb" DESTINATION "${deb_dir}")

  set(armpl_prefix "opt/arm/armpl_${short_version}_gcc")
  set(private_libs amath astring)
  set(data_dir "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-data")
  file(REMOVE_RECURSE "${data_dir}")
  file(MAKE_DIRECTORY "${data_dir}")
  file(GLOB data_archive "${deb_dir}/data.tar.*")
  set(members include lib/libarmpl_lp64.a lib/pkgconfig/armpl-lp64-seq.pc license_terms)
  foreach(lib IN LISTS private_libs)
    list(APPEND members "lib/lib${lib}.a")
  endforeach()
  list(TRANSFORM members PREPEND "./${armpl_prefix}/")
  vcpkg_execute_required_process(
    COMMAND "${CMAKE_COMMAND}" -E tar xf "${data_archive}" ${members}
    WORKING_DIRECTORY "${data_dir}"
    LOGNAME "extract-${TARGET_TRIPLET}-data"
  )
  set(armpl_dir "${data_dir}/${armpl_prefix}")
  set(license_dir "${armpl_dir}/license_terms")
elseif(VCPKG_TARGET_IS_OSX)
  file(GLOB dmg "${archive_dir}/*.dmg")
  set(mount_point "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-mnt")
  set(install_dir "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-install")
  file(REMOVE_RECURSE "${mount_point}" "${install_dir}")
  file(MAKE_DIRECTORY "${mount_point}")
  vcpkg_execute_required_process(
    COMMAND hdiutil attach "${dmg}" -mountpoint "${mount_point}" -readonly -nobrowse
    WORKING_DIRECTORY "${CURRENT_BUILDTREES_DIR}"
    LOGNAME "attach-${TARGET_TRIPLET}"
  )
  file(GLOB installer "${mount_point}/*_install.sh")
  # -y accepts Arm's EULA.
  execute_process(
    COMMAND "${installer}" -y "--install_dir=${install_dir}"
    OUTPUT_FILE "${CURRENT_BUILDTREES_DIR}/install-${TARGET_TRIPLET}-out.log"
    ERROR_FILE "${CURRENT_BUILDTREES_DIR}/install-${TARGET_TRIPLET}-err.log"
    RESULT_VARIABLE install_result
  )
  vcpkg_execute_required_process(
    COMMAND hdiutil detach "${mount_point}"
    WORKING_DIRECTORY "${CURRENT_BUILDTREES_DIR}"
    LOGNAME "detach-${TARGET_TRIPLET}"
  )
  if(NOT install_result EQUAL 0)
    message(FATAL_ERROR "The ArmPL installer failed; see ${CURRENT_BUILDTREES_DIR}/install-${TARGET_TRIPLET}-*.log")
  endif()

  file(GLOB armpl_dir "${install_dir}/armpl_*")
  set(license_dir "${armpl_dir}/license_terms")
  set(private_libs flang_rt.runtime)
elseif(VCPKG_TARGET_IS_WINDOWS)
  file(GLOB armpl_dir "${archive_dir}/*/Arm Performance Libraries/armpl_*")
  set(license_dir "${armpl_dir}/../license_terms")
endif()

file(INSTALL "${armpl_dir}/include/" DESTINATION "${CURRENT_PACKAGES_DIR}/include/armpl")
if(VCPKG_TARGET_IS_WINDOWS)
  set(prefixes "${CURRENT_PACKAGES_DIR}")
  if(NOT VCPKG_BUILD_TYPE)
    list(APPEND prefixes "${CURRENT_PACKAGES_DIR}/debug")
  endif()
  foreach(prefix IN LISTS prefixes)
    file(INSTALL "${armpl_dir}/lib/armpl_lp64.dll.lib" DESTINATION "${prefix}/lib" RENAME armpl_lp64.lib)
    file(INSTALL "${armpl_dir}/bin/armpl_lp64.dll" DESTINATION "${prefix}/bin")
  endforeach()
else()
  foreach(lib IN ITEMS armpl_lp64 ${private_libs})
    file(INSTALL "${armpl_dir}/lib/lib${lib}.a" DESTINATION "${CURRENT_PACKAGES_DIR}/lib")
  endforeach()

  set(pc_file "${CURRENT_PACKAGES_DIR}/lib/pkgconfig/armpl-lp64-seq.pc")
  file(INSTALL "${armpl_dir}/lib/pkgconfig/armpl-lp64-seq.pc" DESTINATION "${CURRENT_PACKAGES_DIR}/lib/pkgconfig")
  vcpkg_replace_string("${pc_file}" "includedir=\${prefix}/include\n" "includedir=\${prefix}/include/armpl\n")
  vcpkg_replace_string("${pc_file}" "-larmpl\n" "-larmpl_lp64\n")
  vcpkg_fixup_pkgconfig()
endif()

vcpkg_install_copyright(FILE_LIST
  "${license_dir}/license_agreement.txt"
  "${license_dir}/third_party_licenses.txt"
)
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
