#!/bin/bash
# 在 reset_station.xml 的灯光那一行后面加一个俯视相机（位置和代码里的 TOP_CAM 一致，默认朝正下方）。可重复运行。
XML=assets/robotstudio_so101/reset_station.xml
if grep -q 'name="top_cam"' "$XML"; then
  echo "top_cam 已经存在"
else
  sed -i '/<light pos="0 0 3.5"/a\    <camera name="top_cam" mode="fixed" pos="0.44 0 1.0" fovy="42"/>' "$XML"
  echo "已加入 top_cam"
fi
grep -n -E 'light pos|top_cam' "$XML"
