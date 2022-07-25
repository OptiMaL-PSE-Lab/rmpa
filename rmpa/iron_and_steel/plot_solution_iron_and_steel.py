import pickle
import pandas as pd 
import numpy as np 
import pandas as pd
import geopandas as gpd
import geoplot as gplt
import matplotlib.pyplot as plt 
import geoplot.crs as gcrs
from shapely.geometry import Polygon, Point
import mapclassify as mc

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
europe=world[world.continent=="Europe"]
europe=europe[(europe.name!="Russia") & (europe.name!="Iceland")]
polygon = Polygon([(-25,35), (40,35), (40,75),(-25,75)])
poly_gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs=world.crs)
europe=gpd.clip(europe, polygon) 

with open('outputs/ellipse_results.pickle','rb') as handle:
	res = pickle.load(handle)

lv = ['nominal_solution','robust_solution']
ln = ['Nominal','Robust']
fig , axs= plt.subplots(1,2,figsize=(6,4),subplot_kw={
	'projection': gcrs.Mercator()
	})
data = pd.read_csv("data/iron_and_steel_csv.csv", skiprows=1)[:33]
plant_names = [i.replace("\xa0", "") for i in data["PLANT"].values[1:]]
plant_emissions = [float(i.replace(",", "")) for i in data["CO2 Emissions per plant (t/yr)"].values[1:]]

vars = list(res[lv[0]].values())[1:]
carbon_tax = vars[:int(len(vars)/2)]
market_share = [v*100 for v in vars[int(len(vars)/2):]]

vars_rob = list(res[lv[1]].values())[1:]
carbon_tax_robust = vars_rob[:int(len(vars)/2)]
market_share_robust = [v*100 for v in vars_rob[int(len(vars)/2):]]

fig,axs = plt.subplots(1,1)
#axs.hist([market_share,market_share_robust],color=['k','r'],alpha=0.5,label=['Nominal','Robust'])
axs.hist(market_share,color='k',alpha=0.75)
axs.set_ylabel('Frequency')
axs.set_xlabel('Market Share of CCS (%)')
axs.ticklabel_format(useOffset=False)
plt.savefig('nominal_hist_ms.pdf')


fig , axs= plt.subplots(1,3,figsize=(9,4),subplot_kw={
	'projection': gcrs.Mercator()
	})
for i in range(2):
	vars = list(res[lv[i]].values())[1:]
	carbon_tax = vars[:int(len(vars)/2)]
	market_share = [v*100 for v in vars[int(len(vars)/2):]]


	locs = pd.read_csv('data/plant_locs.csv')
	names = locs['Plant']
	locations = list(locs['Location'].values)
	for j in range(len(locations)):
		lon_lat = locations[j].split(',')
		locations[j] = Point(reversed([float(l) for l in lon_lat]))


	def scale(minval, maxval):
		def scalar(val):
			return val*20
		return scalar

	geo_res = gpd.GeoDataFrame(data={'Plant':names,'Carbon Tax':carbon_tax, 'Market Share': market_share},geometry=locations)

	

	europe =europe.to_crs("WGS84")
	gplt.polyplot(europe,edgecolor="black",ax=axs[i],projection=gcrs.Mercator())
	axs[i].set_title(ln[i]+' Market Share')
	axs[i].set_ylabel('Latitude')
	axs[i].set_xlabel('Longitude')
	geo_res = geo_res.set_crs("WGS84")
	gplt.pointplot(
	geo_res,
	scale='Market Share',
	#limits=(2, 30),
	legend=True,
	legend_var='scale',
	scale_func=scale,
	color='gray',
	edgecolor='k',
	limits=(0.2, 0.6),
	legend_values=[0.2,0.3,0.4,0.5,0.6],
	legend_labels=['0.2%', '0.3%', '0.4%', '0.5%','0.6%'],
	legend_kwargs={'frameon':False,'markeredgecolor':'k'},
	ax=axs[i]
	)

vars = list(res[lv[0]].values())[1:]
vars_2 = list(res[lv[1]].values())[1:]

carbon_tax = vars[:int(len(vars)/2)]
market_share = [v*100 for v in vars[int(len(vars)/2):]]
carbon_tax_2 = vars_2[:int(len(vars_2)/2)]
market_share_2 = [v*100 for v in vars_2[int(len(vars_2)/2):]]

locs = pd.read_csv('data/plant_locs.csv')
names = locs['Plant']
locations = list(locs['Location'].values)
for j in range(len(locations)):
	lon_lat = locations[j].split(',')
	locations[j] = Point(reversed([float(l) for l in lon_lat]))


def scale_2(minval, maxval):
	def scalar(val):
		return val*60
	return scalar

geo_res = gpd.GeoDataFrame(data={'Emissions':plant_emissions,'Plant':names,'Carbon Tax Difference':np.array(carbon_tax_2)-np.array(carbon_tax), 'Market Share Difference': np.array(market_share_2)-np.array(market_share)},geometry=locations)



europe =europe.to_crs("WGS84")
gplt.polyplot(europe,edgecolor="black",ax=axs[2],projection=gcrs.Mercator())
axs[2].set_title('Market Share Difference')
axs[2].set_ylabel('Latitude')
axs[2].set_xlabel('Longitude')
geo_res = geo_res.set_crs("WGS84")
gplt.pointplot(
geo_res,
scale='Market Share Difference',
#limits=(2, 30),
legend=True,
legend_var='scale',
scale_func=scale_2,
color='gray',
edgecolor='k',
limits=(0,0.1),
legend_values=[0.01,0.04,0.06,0.1],
legend_labels=['0.01%', '0.04%', '0.06%', '0.1%'],
legend_kwargs={'frameon':False,'markeredgecolor':'k'},
ax=axs[2]
)

plt.savefig('outputs/geo_market_share.pdf')


lv = ['nominal_solution','robust_solution']
ln = ['Nominal','Robust']
fig , axs= plt.subplots(1,2,figsize=(6,4),subplot_kw={
	'projection': gcrs.Mercator()
	})
for i in range(2):
	vars = list(res[lv[i]].values())[1:]
	carbon_tax = vars[:int(len(vars)/2)]
	market_share = [v*100 for v in vars[int(len(vars)/2):]]
	locs = pd.read_csv('data/plant_locs.csv')
	names = locs['Plant']
	locations = list(locs['Location'].values)
	for j in range(len(locations)):
		lon_lat = locations[j].split(',')
		locations[j] = Point(reversed([float(l) for l in lon_lat]))


	def scale(minval, maxval):
		def scalar(val):
			return val*0.05
		return scalar

	geo_res = gpd.GeoDataFrame(data={'Plant':names,'Carbon Tax':carbon_tax, 'Market Share': market_share},geometry=locations)

	
	leg_v = carbon_tax[i]
	europe =europe.to_crs("WGS84")
	gplt.polyplot(europe,edgecolor="black",ax=axs[i],projection=gcrs.Mercator())
	axs[i].set_title(ln[i]+' Carbon Tax ($/t)')
	axs[i].set_ylabel('Latitude')
	axs[i].set_xlabel('Longitude')
	geo_res = geo_res.set_crs("WGS84")
	gplt.pointplot(
	geo_res,
	scale='Carbon Tax',
	legend=True,
	legend_var='scale',
	scale_func=scale,
	color='gray',
	edgecolor='black',
	limits=(94, 99),
	legend_values=[leg_v],
	legend_labels=[str(np.round(leg_v,2))],
	legend_kwargs={'frameon':False,'markeredgecolor':'k'},
	ax=axs[i]
	)



plt.savefig('outputs/geo_carbon_tax.pdf')





fig , axs= plt.subplots(1,3,figsize=(9,4),subplot_kw={
	'projection': gcrs.Mercator()
	})


with open('outputs/ellipse_results_01.pickle','rb') as handle:
	res = pickle.load(handle)
vars = list(res[lv[0]].values())[1:]
vars_2 = list(res[lv[1]].values())[1:]

carbon_tax = vars[:int(len(vars)/2)]
market_share = [v*100 for v in vars[int(len(vars)/2):]]
carbon_tax_2 = vars_2[:int(len(vars_2)/2)]
market_share_2 = [v*100 for v in vars_2[int(len(vars_2)/2):]]

locs = pd.read_csv('data/plant_locs.csv')
names = locs['Plant']
locations = list(locs['Location'].values)
for j in range(len(locations)):
	lon_lat = locations[j].split(',')
	locations[j] = Point(reversed([float(l) for l in lon_lat]))


def scale_2(minval, maxval):
	def scalar(val):
		return val*60
	return scalar

geo_res = gpd.GeoDataFrame(data={'Emissions':plant_emissions,'Plant':names,'Carbon Tax Difference':np.array(carbon_tax_2)-np.array(carbon_tax), 'Market Share Difference': np.array(market_share_2)-np.array(market_share)},geometry=locations)
fig.suptitle('Market Share Difference')
europe =europe.to_crs("WGS84")
gplt.polyplot(europe,edgecolor="black",ax=axs[0],projection=gcrs.Mercator())
axs[0].set_title('0.1% Chance Constraints')
axs[0].set_ylabel('Latitude')
axs[0].set_xlabel('Longitude')
geo_res = geo_res.set_crs("WGS84")
gplt.pointplot(
geo_res,
scale='Market Share Difference',
#limits=(2, 30),
cmap='inferno_r',
legend=True,
legend_var='scale',
scale_func=scale_2,
hue='Emissions',
limits=(0,0.1),
legend_values=[0.01,0.02,0.05,0.1],
legend_labels=['0.01%', '0.02%', '0.05%', '0.1%'],
legend_kwargs={'frameon':False},
ax=axs[0]
)





with open('outputs/ellipse_results_1.pickle','rb') as handle:
	res = pickle.load(handle)
vars = list(res[lv[0]].values())[1:]
vars_2 = list(res[lv[1]].values())[1:]

carbon_tax = vars[:int(len(vars)/2)]
market_share = [v*100 for v in vars[int(len(vars)/2):]]
carbon_tax_2 = vars_2[:int(len(vars_2)/2)]
market_share_2 = [v*100 for v in vars_2[int(len(vars_2)/2):]]

locs = pd.read_csv('data/plant_locs.csv')
names = locs['Plant']
locations = list(locs['Location'].values)
for j in range(len(locations)):
	lon_lat = locations[j].split(',')
	locations[j] = Point(reversed([float(l) for l in lon_lat]))


def scale_2(minval, maxval):
	def scalar(val):
		return val*60
	return scalar

geo_res = gpd.GeoDataFrame(data={'Emissions':plant_emissions,'Plant':names,'Carbon Tax Difference':np.array(carbon_tax_2)-np.array(carbon_tax), 'Market Share Difference': np.array(market_share_2)-np.array(market_share)},geometry=locations)

europe =europe.to_crs("WGS84")
gplt.polyplot(europe,edgecolor="black",ax=axs[1],projection=gcrs.Mercator())
axs[1].set_title('1% Chance Constraints')
axs[1].set_ylabel('Latitude')
axs[1].set_xlabel('Longitude')
geo_res = geo_res.set_crs("WGS84")
gplt.pointplot(
geo_res,
scale='Market Share Difference',
#limits=(2, 30),
cmap='inferno_r',
legend=True,
legend_var='scale',
scale_func=scale_2,
hue='Emissions',
limits=(0,0.1),
legend_values=[0.01,0.02,0.05,0.1],
legend_labels=['0.01%', '0.02%', '0.05%', '0.1%'],
legend_kwargs={'frameon':False},
ax=axs[1]
)

with open('outputs/ellipse_results_10.pickle','rb') as handle:
	res = pickle.load(handle)
vars = list(res[lv[0]].values())[1:]
vars_2 = list(res[lv[1]].values())[1:]

carbon_tax = vars[:int(len(vars)/2)]
market_share = [v*100 for v in vars[int(len(vars)/2):]]
carbon_tax_2 = vars_2[:int(len(vars_2)/2)]
market_share_2 = [v*100 for v in vars_2[int(len(vars_2)/2):]]

locs = pd.read_csv('data/plant_locs.csv')
names = locs['Plant']
locations = list(locs['Location'].values)
for j in range(len(locations)):
	lon_lat = locations[j].split(',')
	locations[j] = Point(reversed([float(l) for l in lon_lat]))


def scale_2(minval, maxval):
	def scalar(val):
		return val*60
	return scalar

geo_res = gpd.GeoDataFrame(data={'Emissions':plant_emissions,'Plant':names,'Carbon Tax Difference':np.array(carbon_tax_2)-np.array(carbon_tax), 'Market Share Difference': np.array(market_share_2)-np.array(market_share)},geometry=locations)

europe =europe.to_crs("WGS84")
gplt.polyplot(europe,edgecolor="black",ax=axs[2],projection=gcrs.Mercator())
axs[2].set_title('10% Chance Constraints')
axs[2].set_ylabel('Latitude')
axs[2].set_xlabel('Longitude')
geo_res = geo_res.set_crs("WGS84")
gplt.pointplot(
geo_res,
scale='Market Share Difference',
#limits=(2, 30),
cmap='inferno_r',
legend=True,
legend_var='scale',
scale_func=scale_2,
hue='Emissions',
limits=(0,0.1),
legend_values=[0.01,0.02,0.05,0.1],
legend_labels=['0.01%', '0.02%', '0.05%', '0.1%'],
legend_kwargs={'frameon':False},
ax=axs[2]
)

plt.savefig('outputs/geo_market_share_colored.pdf')
