Data Preprocessing:

Generation Data URL: https://dados.ons.org.br/dataset/geracao-usina-2/resource/cad39843-ad31-42fc-847c-6a643938f621

Weather Data URL: https://ufesbr-my.sharepoint.com/personal/alexandre_xavier_ufes_br/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Falexandre%5Fxavier%5Fufes%5Fbr%2FDocuments%2Fnetcdf%5Ffiles&ga=1

Generator Coordinates URL (the one about capacity): https://dados.ons.org.br/dataset/


Generation Coordinates:
    adjust_id_format.py

Weather Preprocessing:
    To view an individual netcdf file: view_nc_file:
    Extract NETCDF files and concatenate using csv_preprocesing/gen_weather_data.py

Generation Preprocessing:
    Take downloaded file and change file names in gen_data_format.py
    Combine coordinates and concatenate files: match_ids.py

Full Generation Dataset:
    adjust_lat_long.py

To put everything into 1 dataset:
    combine_gen_weather.py