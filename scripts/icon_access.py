# coding: utf-8
import intake
cat = intake.open_catalog('https://digital-earths-global-hackathon.github.io/catalog/catalog.yaml')['UK']
ds = cat['icon_d3hp003'](zoom=7).to_dask()
