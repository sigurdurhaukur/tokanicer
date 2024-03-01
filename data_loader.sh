#bin/bash


if [ ! -d "raw_data" ]; then
  mkdir raw_data
fi

cd raw_data

# 3-4 GB after unzipping
curl --remote-name-all https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/236{/IGC-News1-22.10.TEI.zip}

# ask for permission to unzip

read -p "Do you want to unzip the files? (y/n): " answer
if [[ $answer == "y" ]]; then
  unzip \*.zip
fi

read -p "Do you want to run the preprocessing script? (y/n): " answer
if [[ $answer == "y" ]]; then
  cd ..
  python3 preprocessing.py
fi

echo "Data has been downloaded and preprocessed. You can now run the main script."
