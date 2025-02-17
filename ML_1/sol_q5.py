import numpy as np
import pandas as pd

df = pd.read_csv("./data/taxi_fare_data.csv", quoting=3)

# 불러온 pickup_datetime은 ['2009-06-15 17:26:21 UTC', ...] 과 같은 형태를 지니고 있습니다.
pickup_datetime = df['pickup_datetime'] 

# 우선 연월일('YYYY-MM-DD')와 시간('HH:MM:SS')로 나누어 주고 이를 year_date, time 변수로 각각 넣어줍니다.
year_date = []
time = []

for data in pickup_datetime :
    year_date.append(data.split()[0])
    time.append(data.split()[1])

# 연월일 변수에서 각각의 '연도', '월', '일'을 추출하여 years, months, days 변수에 넣어줍니다.
years = []
months = []
days = []

# 2015-01-07 15:32
for data in year_date :
    years.append(int(data.split('-')[0]))
    months.append(int(data.split('-')[1]))
    days.append(int(data.split('-')[2]))

#시간만 따로 int의 형태로 추출합니다.
hours = [int(i.split(':')[0]) for i in time]

#각 변수의 상위 10개씩만 출력해서 추출이 제대로 되었는지 확인해봅시다.
print(years[:10])
print(months[:10])
print(days[:10])
print(hours[:10])