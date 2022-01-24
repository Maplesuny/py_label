在app當前資料夾輸入:

python -m uvicorn app.main:app --reload

`這串可以有Ip python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload`

輸入Start_time, end_time,montage

`http://127.0.0.1:8000/api/v1/eegData?start_time=0&end_time=1&montage_type=2`
