package main

import (
	"fmt"
	_ "github.com/stretchr/testify/assert"
	"testing"
)

/*
 *  Test Methods
 */

// get html
func Test_getHTML(t *testing.T) {
	//url := "http://www.jtw.zaq.ne.jp/animesong/a/aakue/paipai.html"
	url := "http://www.kasi-time.com/item-24.html"
	html, _ := GetHTML(url)
	fmt.Println(html)
}

// get lyrics from "www.jtw.zaq.ne.jp"
func Test_getUrls_Omakase(t *testing.T) {
	urls := GetUrlsOmakase()
	fmt.Println(urls)
	fmt.Println("歌詞ページの総数:", len(urls))
}

// get lyrics from "www.jtw.zaq.ne.jp"
func Test_getLyrics_Omakase(t *testing.T) {
	url := "http://www.jtw.zaq.ne.jp/animesong/a/aakue/paipai.html"
	lyrics := GetLyricsOmakase(url, -1)
	fmt.Println(lyrics)
}

// get lyrics from "www.kasi-time.com"
/*
func Test_getUrls_KasiTime(t *testing.T) {
	urls := GetUrlsKasiTime()
	fmt.Println(urls)
	fmt.Println("歌詞ページの総数:", len(urls))
}
*/

// get lyrics from "www.kasi-time.com"
func Test_getLyrics_KasiTime(t *testing.T) {
	//url := "http://www.kasi-time.com/item-1.html"
	url := "http://www.kasi-time.com/item-24.html"
	lyrics := GetLyricsKasiTime(url, -1)
	fmt.Println(lyrics)
}
