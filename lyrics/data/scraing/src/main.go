package main

import (
	"fmt"
	"runtime"
	"time"
)

/*
 * defined structs
 */

type LyricsData struct {
	ID        int    `json:"id"`
	Title     string `json:"title"`
	Pronounce string `json:"pronounce"`
	Singer    string `json:"singer"`
	Composer  string `json:"composer"`
	Lyricist  string `json:"lyricist"`
	Arranger  string `json:"arranger"`
	Lyrics    string `json:"lyrics"`
}

type JSONFileFormat struct {
	Data []LyricsData `json:"data"`
}

/*
 * main function
 */
func main() {

	// file path
	var outputJSONFile1 = "../data/omakase.json"
	var outputJSONFile2 = "../data/kasitime.json"

	// variables definition
	var lyrics1 = make([]LyricsData, 0)
	var lyrics2 = make([]LyricsData, 0)

	// get the url of lyrics page
	urls1 := GetUrlsOmakase()
	fmt.Println(urls1)
	urls2 := GetUrlsKasiTime()

	// go routine
	cpus := runtime.NumCPU()
	runtime.GOMAXPROCS(cpus)

	// get lyrics
	// omakase
	for i, v := range urls1 {
		data := GetLyricsOmakase(v, i+1)
		lyrics1 = append(lyrics1, data)
		fmt.Println(data)
		time.Sleep(100 * time.Millisecond)
	}

	jsondata1 := JSONFileFormat{
		Data: lyrics1,
	}
	WriteToJSON(jsondata1, outputJSONFile1)

	// kasitime
	for i, v := range urls2 {
		data := GetLyricsKasiTime(v, i+1)
		lyrics2 = append(lyrics2, data)
		fmt.Println(data)
	}

	jsondata2 := JSONFileFormat{
		Data: lyrics2,
	}
	WriteToJSON(jsondata2, outputJSONFile2)

}
