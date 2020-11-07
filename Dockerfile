FROM armindocachada/tensorflow2-raspberrypi4:2.3.0-cp35-none-linux_armv7l

RUN apt-get update && apt-get -y install unzip
RUN apt-get install -y build-essential
RUN apt-get -y install libjpeg-dev libpng-dev libtiff-dev
RUN apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
RUN apt-get -y install libxvidcore-dev libx264-dev
RUN apt-get install -y python3-dev
RUN apt-get -y install libgtk2.0-dev

ENV OPENCV_VERSION=4.5.0

# building open-cv
WORKDIR /root
RUN wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip
RUN unzip -o $OPENCV_VERSION.zip && unzip opencv_contrib.zip && mv opencv_contrib-$OPENCV_VERSION opencv_contrib
RUN apt-get -y install cmake
WORKDIR /root/opencv-$OPENCV_VERSION/build

RUN cmake -D ENABLE_NEON=ON  -D ENABLE_VFPV3=ON \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
      -D WITH_FFMPEG=ON -D WITH_TBB=ON -D WITH_GTK=ON -D WITH_V4L=ON -D WITH_OPENGL=ON -D WITH_CUBLAS=ON -DWITH_QT=OFF \
      -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES"  --prefix=/usr --extra-version='1~16.04.york0' --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu \
      --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-avresample --disable-filter=resample \
      --enable-avisynth --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca \
      --enable-libcdio --enable-libcodec2 --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme \
      --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus \
      --enable-libpulse --enable-librsvg --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex \
      --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwavpack \
      --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx \
      --enable-openal --enable-opengl --enable-sdl2 --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r \
      --enable-libopencv --enable-libx264 --enable-shared ..

RUN make -j4
RUN make install
RUN ldconfig

WORKDIR /root/
RUN rm -fr /root/opencv* && rm -fr /root/${OPENCV_VERSION}.zip
RUN export READTHEDOCS=True && pip3 install picamera[array]
RUN pip3 install matplotlib && apt-get install python3-tk

CMD ["tail","-f","/dev/null"]