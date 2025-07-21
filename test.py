import os
import json
import base64
import requests
import re
import time
import random
from time import sleep

# 模型服务配置
MODEL_NAME = "qwen2.5-vl-72b-awq"
LOCAL_VLLM_ENDPOINT = 'http://localhost:9076/v1/chat/completions'
IMAGE_PATH = "2.jpg"

# Prompt 配置（请将 prompt2 的 ... 替换成原始内容）
prompt1 = "提取图纸中的如下信息：【几何形状，材料，最小尺寸公差，技术要求&注意事项】\n" \
          "几何形状只需要识别是棱柱形还是圆柱形。\n" \
          "最小尺寸公差需要将上下公差值相减后，取所有差值的最小值，并且不包含形状公差。\n" \
          "技术要求&注意事项如果是其他语种，必须翻译成中文。并且这一条必须是含有文字信息的，如果出现一堆符号或者数字的条目则需要过滤掉。\n" \
          "转成json格式，用中文回答\n"

prompt2 = "你是一个高级机械工程师，将要帮我对于【规格尺寸】信息进行识别。\n" \
        "提取步骤：\n" \
        "1.首先，需要基于零件的形状，判定需要输出的【规格尺寸】形式。\n" \
        "    零件若属于棱柱形，则需要输出【长度】，【宽度】和【高度】，输出形式为【长*宽*高】。\n" \
        "    零件若属于圆柱形，则需要输出【直径】和【高】，输出形式为【直径*高】。\n"\
        "2.然后，基于可能的位置，定位识别出具体的数值。\n"\
        "    可能出现如下几种情况：\n"\
        "    零件为棱柱形的\n"\
        "        3.1若分别从2.1和2.2都识别到了【长】，需要取数值更大的作为【长】。\n"\
        "        3.2若分别从2.3和2.4都识别到了【宽】，需要取数值更大的作为【宽】。\n"\
        "        3.3若分别从2.5和2.6都识别到了【高】，需要取数值更大的作为【高】。\n"\
        "    零件为圆柱形的\n"\
        "        3.1直径参数往往在其前面会带有一个Φ符号，需要先对于直径参数进行识别，这一步一定要尽可能多的进行识别。\n"\
        "        3.2若识别到多个直径参数，需要取数值最大的作为【直径】，在这一步比较大小的时候，需要注意小数点的问题，170是比84.8大的。\n"\
        "4.最终，输出正确形式的【规格尺寸】信息，并输出为json的字典格式，只用中文回答，不要输出其他内容。\n"\
        "    信息包含如下信息，包含的是【】中包含的文本代表的信息而非数字1.1等内容：\n"\
        "   1.假如是棱柱形：\n"\
        "       1.1【零件长度】\n"\
        "       1.2【俯/仰视图长度】\n"\
        "       1.3【主/后视图长度】\n"\
        "       1.4【零件宽度】\n"\
        "       1.5【俯/仰视图宽度】\n"\
        "       1.6【左/右视图宽度】\n"\
        "       1.7【零件高度】\n"\
        "       1.8【左/右视图高度】\n"\
        "       1.9【主/后视图高度】\n"\
        "   2.假如是圆柱形：\n"\
        "       2.1【零件直径】\n"\
        "       2.2【俯/仰视图直径】\n"\
        "       2.3【圆柱侧视图直径】\n"\
        "       2.4【零件高度】\n"\
        "       2.5【圆柱侧视图高度】\n"\
        "任何1.1-2.5的参数，识别不出来，直接输出无。\n"

prompt3 = """零件加工工艺大类和细分类别：
1.【通用CNC机加工】包含【车削】【铣削】【钻孔】【磨削】【倒角】
2.【特种CNC机加工】包含【龙门机加工】【多轴CNC】【齿轮加工】
3.【注塑成型】包含【热塑性注塑】【双料注塑】【气辅注塑】
4.【粉末冶金】包含【粉末压制烧结】【热等静压（HIP）】【注射成型】
5.【3D打印】包含【3D打印】
6.【锻造】包含【热锻】
7.【铸造】包含【砂模铸造】【压铸】【蜡模铸造】
9.【喷漆工艺】包含【喷漆】【烤漆】
10.【焊接】包含【焊接】
11.【钣金加工】包含【激光切割】【折弯】【冲压】
12.【挤铝】包含【铝型材挤压】
13.【线加工】包含【电火花线切割】【快速走丝】【慢走丝】
14.【标准件加工】包含【螺丝】【螺母】
""" + "提取图纸中的如下信息：\n" \
     "零件加工工艺：只输出最可能使用到的工艺大类\n" \
     "转成json格式，用中文回答\n"

prompt_map = {
    "几何形状/材料/公差": prompt1,
    "规格尺寸": prompt2,
    "加工工艺": prompt3
}


def call_vllm(image_path, prompt):
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    image_data_uri = f"data:image/jpeg;base64,{image_b64}"

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "temperature": 0.0,
        "top_p": 0.1
    }

    response = requests.post(LOCAL_VLLM_ENDPOINT, json=payload)
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]

    match = re.findall(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if match:
        content = match[-1]

    try:
        return json.loads(content)
    except:
        return {"解析失败": content}


def merge_results(task_outputs):
    merged = {}
    for result in task_outputs.values():
        if isinstance(result, dict):
            merged.update(result)
    return merged


def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"找不到图片文件：{IMAGE_PATH}")
        return

    all_results = []
    all_durations = []

    for i in range(10):
        start = time.time()

        task_outputs = {}
        for task_name, prompt in prompt_map.items():
            task_outputs[task_name] = call_vllm(IMAGE_PATH, prompt)

        end = time.time()
        duration_ms = (end - start) * 1000
        all_durations.append(duration_ms)
        all_results.append(merge_results(task_outputs))

        if i < 9:
            sleep(1)  # 间隔1秒（最后一次不需要间隔）

    avg_duration = sum(all_durations) / len(all_durations)
    random_result = random.choice(all_results)

    print(json.dumps(random_result, ensure_ascii=False, indent=2))
    print(f"\n平均耗时: {avg_duration:.2f} 毫秒")


if __name__ == "__main__":
    main()
