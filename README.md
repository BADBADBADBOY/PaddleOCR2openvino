### PaddleOCR 转成 Openvino

### 依赖环境

去百度的AIstudio（https://aistudio.baidu.com/overview），选择免费的CPU环境创建一个项目即可，将这个项目代码拷贝到里面去。

```
 pip install -r requirements.txt --user
```

### 模型转换

```
python prepare.py make
```


### 结果测试

运行paddle

```
sh test_pp.sh
```

运行onnx
```
sh test_pp_onnx.sh
```

运行openvino
```
python test_serviceOCRModule.py
```

转换完成后，如果你在CPU上做识别，serviceOCRModule这个模块放到任何的项目中去做识别。还是比较方便的，openvino在cpu上加速还是不错的。

### 如果你想转移项目，可以先清除项目中的模型和垃圾文件
```
python prepare.py clear
```


### 参考文献

https://github.com/PaddlePaddle/PaddleOCR