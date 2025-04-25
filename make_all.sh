alias make="make -j$(nproc)"

mkdir bin
mkdir bin/hotpants
mkdir bin/bach
mkdir bin/xbach
mkdir bin/emxbach

ln -sf ../../../BACH/cl_kern bin/bach/cl_kern
ln -sf ../../../XBACH/cl_kern bin/xbach/cl_kern
ln -sf ../../cl_kern bin/emxbach/cl_kern

cd ../hotpants
echo "building hotpants"
(make clean && make && cp hotpants ../Thesis-EMX-BACH/bin/hotpants/hotpants) > /dev/null 2>&1
echo built hotpants with code $?

cd ../BACH
echo "building bach"
(make clean && make && cp BACH ../Thesis-EMX-BACH/bin/bach/bach) > /dev/null 2>&1 
echo built bach with code $?

cd ../XBACH
echo "building xbach"
(make clean && make > /dev/null 2>&1 && cp XBACH ../Thesis-EMX-BACH/bin/xbach/xbach) > /dev/null 2>&1
echo built xbach with code $?

cd ../Thesis-EMX-BACH
echo "building emxbach"
(make clean && make > /dev/null 2>&1 && cp EMXBACH ./bin/emxbach/emxbach) > /dev/null 2>&1
echo built emxbach with code $?
