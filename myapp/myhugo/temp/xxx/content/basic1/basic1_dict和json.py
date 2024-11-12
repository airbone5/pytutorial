import json
# 字典
dict = {'A':'123R','B':'678S'}
print(dict['A'])
# 輸出: 123R

# some JSON:
a = '{ "name":"Rahul", "age":21, "city":"Banglore"}'
# parse x:

b = json.loads(a)
print(b["city"])


# Nested JSON String
nested_json = {"person": {"name": "Alice", "age": 25, "address": {"city": "Wonderland", "country": "Fictional"}}}

# Print the nested JSON string
print("Nested JSON String:", nested_json)
print(nested_json['person'])

#❓為甚麼出錯
# nested_json = '{"person": {"name": "Alice", "age": 25, "address": {"city": "Wonderland", "country": "Fictional"}}}'
# print(nested_json['person'])
# 結果 
#     print(nested_json['person'])
#           ~~~~~~~~~~~^^^^^^^^^^
# TypeError: string indices must be integers, not 'str'



dict = {
"color": "blue",
"car": "farari",
"flower": "jasmine"
}
print(dict)
