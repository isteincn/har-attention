train_files='subject101.dat subject102.dat subject103.dat subject104.dat subject107.dat subject108.dat subject109.dat'
val_file="subject105.dat"
test_file="subject106.dat"

rm -f train.dat
rm -f test.dat
rm -f validate.dat

for f in $train_files; do
	echo "Merging " $f
	awk -f clean.awk $f >> train.dat
done

awk -f clean.awk $val_file >> validate.dat
awk -f clean.awk $test_file >> test.dat

