DROP TABLE dataset_yelp
CREATE TABLE yelp_project.dbo.dataset_yelp(
  name  varchar(MAX),
  total_rating float,
  category varchar(MAX),
  price_category varchar(MAX),
  number_reviews float,
  inspec_period float,
  period_rating float,
  review_text varchar(MAX),
  number_inspections int,
  health_score int,
  number_violations float,
  inspec_type varchar(MAX),
  inspec_vio varchar(MAX),
  verdict varchar(MAX)
)

