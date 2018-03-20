package main

import (
	"github.com/PuerkitoBio/goquery"
	"regexp"
	"strconv"
)

/*
 * defined functions
 */

// get url from "www.jtw.zaq.ne.jp"
func GetUrlsOmakase() []string {
	topPageUrl := "http://www.jtw.zaq.ne.jp/animesong/"
	reg1 := regexp.MustCompile(`^0/.*`)
	reg2 := regexp.MustCompile(`mailto:`)
	reg3 := regexp.MustCompile(`^(http://.+/).*$`)
	reg4 := regexp.MustCompile(`\.\./`)

	indexUrls := make([]string, 0)
	animeIndexUrls := make([]string, 0)
	lyricsPageUrls := make([]string, 0)

	// get index urls
	doc, _ := goquery.NewDocument(topPageUrl)
	doc.Find("a").Each(func(_ int, s *goquery.Selection) {
		url, _ := s.Attr("href")
		if !reg1.MatchString(url) && !reg2.MatchString(url) {
			indexUrls = append(indexUrls, topPageUrl+url)
		}
	})
	indexUrls = indexUrls[1 : len(indexUrls)-5]

	// get anime index urls
	for _, v := range indexUrls {
		baseUrl := ""
		if reg3.MatchString(v) {
			group := reg3.FindStringSubmatch(v)
			baseUrl = group[1]
		}

		doc, _ := goquery.NewDocument(v)
		doc.Find("a").Each(func(_ int, s *goquery.Selection) {
			url, _ := s.Attr("href")
			if url != "" && !reg3.MatchString(url) && !reg4.MatchString(url) {
				animeIndexUrls = append(animeIndexUrls, baseUrl+url)
			}
		})
	}

	// get lyrics page urls
	for _, v := range animeIndexUrls {
		baseUrl := ""
		if reg3.MatchString(v) {
			group := reg3.FindStringSubmatch(v)
			baseUrl = group[1]
		}

		doc, _ := goquery.NewDocument(v)
		doc.Find("a").Each(func(_ int, s *goquery.Selection) {
			url, _ := s.Attr("href")
			if url != "" && !reg3.MatchString(url) && !reg4.MatchString(url) {
				lyricsPageUrls = append(lyricsPageUrls, baseUrl+url)
			}
		})

	}

	return lyricsPageUrls
}

// get url from "www.kasi-time.com"
func GetUrlsKasiTime() []string {
	baseUrl := "http://www.kasi-time.com/item-"
	pageUrls := make([]string, 0)

	errCount := 0
	count := 74410
	for errCount < 5 {
		url := baseUrl + strconv.Itoa(count) + ".html"
		html, err := GetHTML(url)

		if html == "" {
			break
		}

		if err != nil {
			errCount++
			continue
		} else {
			count++
			errCount = 0
		}
	}

	for i := 1; i <= count; i++ {
		url := baseUrl + strconv.Itoa(i) + ".html"
		pageUrls = append(pageUrls, url)
	}

	return pageUrls
}
