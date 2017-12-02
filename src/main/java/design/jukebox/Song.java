package design.jukebox;

/**
 * Created by tao on 10/24/17.
 */
public class Song {

    public String getSongName() {
        return songName;
    }

    public void setSongName(String songName) {
        this.songName = songName;
    }

    private String songName;
    public String toString(){
        return songName;
    }
}
