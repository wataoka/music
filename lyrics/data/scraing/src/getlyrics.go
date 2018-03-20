package main

import (
	"github.com/kennygrant/sanitize"
	"regexp"
)

/*
 * defined functions
 */

// get lyrics from "www.jtw.zaq.ne.jp"
func GetLyricsOmakase(url string, id int) LyricsData {
	regTitle := regexp.MustCompile("(?is)<title>(.*?)</title>")
	regInfo := regexp.MustCompile("(?is).*?作詞[:：](.*?)／作曲[:：](.*?)／編曲[:：](.*?)／\n歌[:：](.*?)\n\n(.*?)</pre></td>")

	title := ""
	singer := ""
	lyricist := ""
	composer := ""
	arranger := ""
	lyrics := ""

	errCount := 0
	html := ""
	for errCount < 5 {
		html1, err := GetHTML(url)
		html = html1
		if err != nil {
			errCount++
		} else {
			html, _ = sjis_to_utf8(html)
			break
		}
	}

	if regTitle.MatchString(html) {
		group := regTitle.FindStringSubmatch(html)
		title = group[1]
	}

	if regInfo.MatchString(html) {
		group := regInfo.FindStringSubmatch(html)
		lyricist = group[1]
		composer = group[2]
		arranger = group[3]
		singer = group[4]
		lyrics = group[5]
	}

	data := LyricsData{
		ID:       id,
		Title:    title,
		Lyricist: lyricist,
		Composer: composer,
		Arranger: arranger,
		Singer:   singer,
		Lyrics:   lyrics,
	}

	return data
}

// get lyrics from "www.kasi-time.com"
func GetLyricsKasiTime(url string, id int) LyricsData {

	regUrl := regexp.MustCompile(`www.kasi-time.com/item-(.*?).html`)
	regTitle := regexp.MustCompile(`(?is)<div id="song_info_table">.*?<h1>(.*?)</h1>`)
	regPronounce := regexp.MustCompile(`(?is)<td class="td2">読み</td>.*?<td>(.*?)</td>`)
	regInfo := regexp.MustCompile(`(?is)<meta name="description" content="歌手:(.*?)[ 　]+作詞:(.*?)[ 　]+作曲:(.*?)[ 　]+.*?>`)
	regArranger := regexp.MustCompile(`(?is)<td class="td1">編曲</td>.*?<td>(.*?)</td>`)
	regLyrics := regexp.MustCompile(`document.write\('(.+)'\);`)

	baseUrl := "http://www.kasi-time.com/item_js.php?no="
	pageUrl := baseUrl

	title := ""
	pronounce := ""
	singer := ""
	lyricist := ""
	composer := ""
	arranger := ""
	lyrics := ""

	if regUrl.MatchString(url) {
		group := regUrl.FindStringSubmatch(url)
		pageID := group[1]
		pageUrl += pageID
	}

	html1 := ""
	html2 := ""
	errCount1 := 0
	errCount2 := 0

	for errCount1 < 5 {
		html, err := GetHTML(url)
		html1 = html
		if err != nil {
			errCount1++
		} else {
			break
		}
	}

	for errCount2 < 5 {
		html, err := GetHTML(pageUrl)
		html2 = html
		if err != nil {
			errCount1++
		} else {
			break
		}
	}

	if regTitle.MatchString(html1) {
		group := regTitle.FindStringSubmatch(html1)
		title = group[1]
	}

	if regPronounce.MatchString(html1) {
		group := regPronounce.FindStringSubmatch(html1)
		pronounce = group[1]
	}

	if regInfo.MatchString(html1) {
		group := regInfo.FindStringSubmatch(html1)
		singer = group[1]
		lyricist = group[2]
		composer = group[3]
	}

	if regArranger.MatchString(html1) {
		group := regArranger.FindStringSubmatch(html1)
		arranger = sanitize.HTML(group[1])
	}

	if regLyrics.MatchString(html2) {
		group := regLyrics.FindStringSubmatch(html2)
		lyrics = sanitize.HTML(group[1])
	}

	data := LyricsData{
		ID:        id,
		Title:     title,
		Pronounce: pronounce,
		Singer:    singer,
		Lyricist:  lyricist,
		Composer:  composer,
		Arranger:  arranger,
		Lyrics:    lyrics,
	}

	return data
}
