


import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET


def arvix_search(query: str,  max_results: int = 5) -> str:
    papers = []
    result = ""
    params = {"search_query": f"all:{query}",  "start": 0, "max_results": max_results,"sortBy": "relevance",
             "sortOrder": "descending"}

    url = ("http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params))
    response = urllib.request.urlopen(url)
    xml_data = response.read()
    root = ET.fromstring(xml_data)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}

    for entry in root.findall("atom:entry", namespace):
        title = entry.find("atom:title",namespace).text.strip()
        abstract = entry.find( "atom:summary", namespace).text.strip()
        published = entry.find( "atom:published", namespace ).text
        arxiv_url = entry.find( "atom:id", namespace ).text
        authors = [author.find("atom:name",namespace).text for author in entry.findall("atom:author",namespace)]
        papers.append({"title": title,"abstract": abstract, "authors": authors, "published": published, "url": arxiv_url,})

    for paper in papers[:max_results]:
        result += f"""Title: {paper['title']}
                      URL: {paper['url']}
                      Abstract: {paper['abstract']}
                  """

    print("result length:", len(result))
    return result
