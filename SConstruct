# Create the build environment.
build_env = Environment(CPPPATH = ['/home/thomas/Projects/ML/caffe/distribute/include',
                                   '/usr/local/cuda-7.0/include'],
                        tools = ['default'])
build_env.Append(CCFLAGS = ['-Wall', '-Wextra'])

# Build the demo.
build_env.Program('src/nnet.cpp', LIBS = ['caffe',
                                          'glog',
                                          'opencv_core',
                                          'opencv_highgui',
                                          'opencv_imgproc',
                                          'opencv_video',
                                          'protoc'],
                                  LIBPATH = ['/home/thomas/Projects/ML/caffe/build/lib'])