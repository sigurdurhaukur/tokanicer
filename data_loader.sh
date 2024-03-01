#bin/bash


if [ ! -d "raw_data" ]; then
  mkdir raw_data
fi

cd raw_data

curl --remote-name-all https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/236{/IGC-News1-22.10.TEI.zip}
