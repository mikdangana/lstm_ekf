#!/bin/sh
host=34.219.25.234
type=c4large
apphost=18.236.147.70

cd ~/woodside/lstm_ekf
rm *.pickle
rm -fr run_*
( for i in {1..1000000}; do curl 'http://$apphost:8878/fetchremote?q=' 1>/dev/null 2>/dev/null; done ) &

for x in 3 10 20; do 
  make steps iter=$x; 
  for i in {1..500}; do 
    curl 'http://$apphost:8878/removesearchendpoint' 1>/dev/null 2>/dev/null; 
    curl 'http://$apphost:8878/removedbendpoint' 1>/dev/null 2>/dev/null; 
  done; 
done

tarball="steps$type$host.tar"
tar cvf $tarball run_*;
scp -i workkeypair.pem $tarball ubuntu@34.214.183.176:~/woodside/lstm_ekf/
