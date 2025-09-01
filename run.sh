docker build -t curvas2025 .

chmod -R 777 ./test_output
docker run --rm \
    -v ./test_input:/input \
    -v ./test_output:/output \
    curvas2025
