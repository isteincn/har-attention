train_files='subject101.dat subject102.dat subject103.dat subject104.dat subject107.dat subject108.dat subject109.dat'
val_file="subject105.dat"
test_file="subject106.dat"

rm -f train.dat

for f in $train_files; do
	echo "Merging " $f
	awk '$2!=0 {print $1, $2, $5, $6, $7, $22, $23, $24, $39, $40, $41}' $f >> train.dat
done

awk '$2!=0 {print $1, $2, $5, $6, $7, $22, $23, $24, $39, $40, $41}' $val_file >> validate.dat
awk '$2!=0 {print $1, $2, $5, $6, $7, $22, $23, $24, $39, $40, $41}' $test_file >> test.dat
