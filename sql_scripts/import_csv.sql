BULK INSERT yelp_project.dbo.dataset_yelp
FROM 'YOUR PATH TO CSV'
WITH
(
  FIRSTROW = 2,
  FIELDTERMINATOR = '|',  --CSV field delimiter
  ROWTERMINATOR = '\n',   --Use to shift the control to next row
  TABLOCK
)
