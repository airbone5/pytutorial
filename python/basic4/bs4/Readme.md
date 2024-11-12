beautifulsoup4(bs4)需要另外安裝 [buildtools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
簡單來說就是把網頁變成DOM

範例
1. 找出網頁中每個水庫的有效蓄水量

## 問題
1. 為甚麼SVG拿到的是空的,例如[程式碼](bs4_test3.py)
    - 提示:
        - chrome SOURCE tab 中的index.html 中,也是顯示空的
    - 答案
        - 因為,內容是由javascript畫出來的 ,進一步的解決辦法可以參考[這裡](https://stackoverflow.com/questions/70662237/how-to-scrape-svg-element-from-a-website-using-beautiful-soup)
