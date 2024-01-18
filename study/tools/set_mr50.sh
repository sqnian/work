# ./power-tool --dev=/dev/aip_bi0 --vddcorevolt=800 --sclk=1000
# ./power-tool --dev=/dev/aip_bi1 --vddcorevolt=800 --sclk=1000
# ./power-tool --dev=/dev/aip_bi2 --vddcorevolt=800 --sclk=1000
# ./power-tool --dev=/dev/aip_bi3 --vddcorevolt=800 --sclk=1000
# ./power-tool --dev=/dev/aip_bi4 --vddcorevolt=800 --sclk=1000
# ./power-tool --dev=/dev/aip_bi5 --vddcorevolt=800 --sclk=1000
# ./power-tool --dev=/dev/aip_bi6 --vddcorevolt=800 --sclk=1000
# ./power-tool --dev=/dev/aip_bi7 --vddcorevolt=800 --sclk=1000


prefix="iluvatar" # 3.1.0 版本使用该前缀

cards=( 0 1 2 3 4 5 6 7)
for num in ${cards[@]}; do
    ./power-tool --dev=/dev/${prefix}${num} --vddcorevolt=800 --sclk=1000
done