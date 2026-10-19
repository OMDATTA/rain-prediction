# Data

Historical training data comes from the Kaggle dataset
[Rain in Australia (weatherAUS)](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package):
about 145k daily observations from 49 weather stations, 2007–2017.

The full CSV is not committed (Kaggle licence and size). Download it with **one** of:

```bash
# 1. kagglehub (no API key needed for public datasets in Colab)
python -c "import kagglehub; print(kagglehub.dataset_download('jsphyg/weather-dataset-rattle-package'))"

# 2. Kaggle CLI (needs ~/.kaggle/kaggle.json)
kaggle datasets download -d jsphyg/weather-dataset-rattle-package -p data --unzip
```

Then place `weatherAUS.csv` in this folder. `sample_weather.csv` is a 300-row sample used by the tests and CI.

Live and historical weather for any location comes from the free [Open-Meteo](https://open-meteo.com/) forecast and archive APIs. No key is required.
