import json

def get_current_weather(location, unit="fahrenheit"):
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": unit}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]
# region コメント
    # name: 関数名
    # description: 関数の説明
    # parameters: 関数の引数
    # type: 引数の型
    # properties: 引数のプロパティ
    # enum: 引数の列挙型
    # required: 引数の必須項目
# endregion

from openai import OpenAI

client = OpenAI()

messages = [
    {"role": "user", "content": "東京の天気はどうですか？"},
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)
print(response.to_json(indent=2))

# region コメント

    # 出力サンプル
    # {
    #   "id": "chatcmpl-BYqy1L6vBHi2b8I5DsUw7VTwxPaLH",
    #   "choices": [
    #     {
    #       "finish_reason": "tool_calls",
    #       "index": 0,
    #       "logprobs": null,
    #       【今までここにLLMが生成したメッセージが含まれていた】
    #       "message": {
    #         "content": null,
    #         "refusal": null,
    #         "role": "assistant",
    #         "annotations": [],
    #         【ここにLLMが実行したいToolの内容が記載されている。】
    #         "tool_calls": [
    #           {
    #             "id": "call_2hwWMV1JXwEL6bIrdrsy4CEd",
    #             "function": {
    #               "arguments": "{\"location\":\"Tokyo, JP\"}",
    #               "name": "get_current_weather"
    #             },
    #             "type": "function"
    #           }
    #         ]
    #       }
    #     }
    #   ],
    #   "created": 1747646457,
    #   "model": "gpt-4o-2024-08-06",
    #   "object": "chat.completion",
    #   "service_tier": "default",
    #   "system_fingerprint": "fp_90122d973c",
    #   "usage": {
    #     "completion_tokens": 17,
    #     "prompt_tokens": 81,
    #     "total_tokens": 98,
    #     "completion_tokens_details": {
    #       "accepted_prediction_tokens": 0,
    #       "audio_tokens": 0,
    #       "reasoning_tokens": 0,
    #       "rejected_prediction_tokens": 0
    #     },
    #     "prompt_tokens_details": {
    #       "audio_tokens": 0,
    #       "cached_tokens": 0
    #     }
    #   }
    # }

    # tool_calls の中身

    # id: "call_2hwWMV1JXwEL6bIrdrsy4CEd"
    # └ このツールコール自体に割り振られた一意ID。

    # type: "function"
    # └ 関数呼び出しであることを示す。

    # function:

    # name: "get_current_weather" → 呼び出す関数名。

    # arguments: "{"location":"Tokyo, JP"}" → 関数に渡す引数を JSON 文字列化したもの。

# endregion

response_message = response.choices[0].message
messages.append(response_message.to_dict())

available_functions = {
    "get_current_weather": get_current_weather,
}

# 使いたい関数は複数あるかもしれないのでループ
for tool_call in response_message.tool_calls:
    # 関数を実行
    function_name = tool_call.function.name
    function_to_call = available_functions[function_name]
    function_args = json.loads(tool_call.function.arguments)
    function_response = function_to_call(
        location=function_args.get("location"),
        unit=function_args.get("unit"),
    )
    print(function_response)

    # 関数の実行結果を会話履歴としてmessagesに追加
    messages.append(
        {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_response,
        }
    )

    print(json.dumps(messages, ensure_ascii=False, indent=2))
    
    # 再度APIにリクエスト送信
    second_response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )
    
    print(second_response.to_json(indent=2))