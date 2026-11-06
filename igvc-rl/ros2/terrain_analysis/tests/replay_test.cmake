execute_process(COMMAND "${REPLAY}" "${SOURCE_DIR}/tests/replay_input.csv" "${BINARY_DIR}/replay_output.csv"
  RESULT_VARIABLE status ERROR_VARIABLE error)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "replay failed: ${error}")
endif()
file(READ "${BINARY_DIR}/replay_output.csv" actual)
if(NOT actual MATCHES "1,1,[^,]+,[^,]+,2,2,0,ground" OR
   NOT actual MATCHES "obstacle" OR NOT actual MATCHES "4,,,,,,,empty")
  message(FATAL_ERROR "replay did not contain expected ground, obstacle and expired frame: ${actual}")
endif()
execute_process(COMMAND "${REPLAY}" "${SOURCE_DIR}/tests/replay_invalid.csv" "${BINARY_DIR}/replay_invalid_output.csv"
  RESULT_VARIABLE bad_status)
if(bad_status EQUAL 0)
  message(FATAL_ERROR "replay accepted backward time")
endif()
file(READ "${SOURCE_DIR}/tests/replay_input.csv" original)
file(WRITE "${BINARY_DIR}/alias_input.csv" "${original}")
execute_process(COMMAND "${REPLAY}" "${BINARY_DIR}/alias_input.csv" "${BINARY_DIR}/./alias_input.csv"
  RESULT_VARIABLE alias_status)
file(READ "${BINARY_DIR}/alias_input.csv" after)
if(alias_status EQUAL 0 OR NOT after STREQUAL original)
  message(FATAL_ERROR "replay failed to preserve aliased input")
endif()
