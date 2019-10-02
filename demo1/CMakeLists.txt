# create an executable
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${DEMO_APPLICATION_BINARY_DIR}/demo1)
ADD_EXECUTABLE (demo1controller main2.cpp)
ADD_EXECUTABLE (simviz01 simviz.cpp)

# and link the library against the executable
TARGET_LINK_LIBRARIES (demo1controller ${DEMO_APPLICATION_COMMON_LIBRARIES})
TARGET_LINK_LIBRARIES (simviz01 ${DEMO_APPLICATION_COMMON_LIBRARIES})

# export resources such as model files.
# NOTE: this requires an install build
SET(APP_RESOURCE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources)
FILE(MAKE_DIRECTORY ${APP_RESOURCE_DIR})

FILE(COPY world.urdf ../resources/panda_arm_hand.urdf DESTINATION ${APP_RESOURCE_DIR})

FILE(COPY world.urdf ../resources/base_link.obj DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link_0.0.obj DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link_1.0.obj DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link_2.0.obj DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link_3.0.obj DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link_3.0_tip.obj DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link_12.0_right.obj DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link_13.0.obj DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link_14.0.obj DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link_15.0.obj DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link_15.0_tip.obj DESTINATION ${APP_RESOURCE_DIR})

FILE(COPY world.urdf ../resources/link0.obj DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link1.obj DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link2.obj DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link3.obj DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link4.obj DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link5.obj DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link6.obj DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link7.obj DESTINATION ${APP_RESOURCE_DIR})

FILE(COPY world.urdf ../resources/link0.dae DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link1.dae DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link2.dae DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link3.dae DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link4.dae DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link5.dae DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link6.dae DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link7.dae DESTINATION ${APP_RESOURCE_DIR})

FILE(COPY world.urdf ../resources/link0.mtl DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link1.mtl DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link2.mtl DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link3.mtl DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link4.mtl DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link5.mtl DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link6.mtl DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY world.urdf ../resources/link7.mtl DESTINATION ${APP_RESOURCE_DIR})
