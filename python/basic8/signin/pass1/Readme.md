


## 程式提要
- [基本(server_empty.py)](./server_empty.py)
- [使用dotenv(server_dotenv.py)](./server_dotenv.py)
    1. 套件
        ```
        pip install python-dotenv
        ```
    1. 把`.env_your_secret` 複製成另一個檔案`.env` 並且填入id,secret
- 預計其他檔案,都改自server_dotenv.py  
- [server_1_list](./server_1_list.py)  這裡增加了一個API,http://localhost:5000/list 
    - 但是問題是,沒有security