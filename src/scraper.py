import requests
import pandas as pd
import os
from tqdm import tqdm
from bs4 import BeautifulSoup


class LinkedinScraper:
  def __init__(self, location, keywords=None, amount=50):
    self.location = location
    self.keywords = keywords
    self.amount = amount
    self.job_ids = []
    self.jobs = []

    if amount > 1000:
      print("⚠️ WARNING: LinkedIn only allows you to scrape 1000 jobs per search. ⚠️")
      print("⚠️ WARNING: The amount will be set to 1000. ⚠️")
      self.amount = 1000
    if keywords == None:
      self.all_jobs_url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?location={self.location}"
      self.all_jobs_url += "&start={}"
    else:
      self.all_jobs_url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={self.keywords}&location={self.location}"
      self.all_jobs_url += "&start={}"

    self.job_url = "https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{}"

  def save_to_csv(self, filename="data/raw/jobs.csv"):
    print("📝 Saving jobs to CSV file...")
    if os.path.isfile(filename):
      existing_ids = set(pd.read_csv(filename, sep=";")["id"])
    else:
      existing_ids = set()

    # Filter out jobs that are already saved in the CSV

    unique_jobs = [job for job in self.jobs if int(
        job["id"]) not in existing_ids]

    if unique_jobs:
      df = pd.DataFrame(unique_jobs)
      df.to_csv(filename, mode='a', sep=';',
                header=not os.path.isfile(filename), index=False)

  def _get_job_ids(self):
    for i in tqdm(range(0, self.amount, 25), desc="💼 Scraping job IDs 🔍", ascii=True, colour="#0077B5"):
      res = requests.get(self.all_jobs_url.format(i))
      soup = BeautifulSoup(res.text, 'html.parser')
      alljobs_on_this_page = soup.find_all("li")
      for x in range(0, len(alljobs_on_this_page)):
        try:
          jobid = alljobs_on_this_page[x].find(
              "div", {"class": "base-card"}).get('data-entity-urn').split(":")[3]
          self.job_ids.append(jobid)
        except:
          print("❌ One Job ID could not be retrieved ❌")
          pass

  def scrape(self):
    # First scrape the job ids
    self._get_job_ids()

    # Then scrape the job details
    for j in tqdm(range(0, len(self.job_ids)), desc="💼 Scraping job details 🔍", ascii=True, colour="#0077B5"):
      job = {}  # Create a new job dictionary
      resp = requests.get(self.job_url.format(self.job_ids[j]))
      soup = BeautifulSoup(resp.text, 'html.parser')

      job["id"] = self.job_ids[j]
      job["date_scraped"] = pd.Timestamp.now()
      job["keyword_scraped"] = self.keywords
      job["location_scraped"] = self.location
      job["linkedin_num"] = j

      try:
        job["company"] = soup.find(
            "div", {"class": "top-card-layout__card"}).find("a").find("img").get('alt')
      except:
        job["company"] = None

      try:
        job["title"] = soup.find(
            "div", {"class": "top-card-layout__entity-info"}).find("a").text.strip()
      except:
        job["title"] = None

      try:
        job["num_applicants"] = soup.find(
            "div", {"class": "top-card-layout__entity-info"}).find("h4").find("span", {"class": "num-applicants__caption"}).text.strip()
      except:
        job["num_applicants"] = None

      try:
        job["date_posted"] = soup.find(
            "div", {"class": "top-card-layout__entity-info"}).find("h4").find("span", {"class": "posted-time-ago__text"}).text.strip()
      except:
        job["date_posted"] = None

      try:
        ul_element = soup.find(
            "ul", {"class": "description__job-criteria-list"})

        for li_element in ul_element.find_all("li"):
          subheader = li_element.find(
              "h3", {"class": "description__job-criteria-subheader"}).text.strip()
          criteria = li_element.find("span", {
              "class": "description__job-criteria-text description__job-criteria-text--criteria"}).text.strip()

          if "Seniority level" in subheader:
            job["level"] = criteria
          elif "Employment type" in subheader:
            job["employment_type"] = criteria
          elif "Job function" in subheader:
            job["function"] = criteria
          elif "Industries" in subheader:
            job["industries"] = criteria
      except:
        job["level"] = None
        job["employment_type"] = None
        job["function"] = None
        job["industries"] = None

      try:
        job["description"] = soup.find(
            "div", {"class": "description__text description__text--rich"}).text.strip()
      except:
        job["description"] = None

      self.jobs.append(job)

      # Checkpoint to save the jobs to the CSV file every 500 jobs
      if (j + 1) % 500 == 0:
        self.save_to_csv()
        self.jobs = []

    if self.jobs:
      self.save_to_csv()


if __name__ == "__main__":
  scraper = LinkedinScraper(location="Taiwan", amount=1000)
  scraper.scrape()
