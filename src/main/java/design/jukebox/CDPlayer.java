package design.jukebox;

/**
 * Created by tao on 10/24/17.
 */
public class CDPlayer {

    private PlayList P;
    private CD c;


    public PlayList getP() {
        return P;
    }

    public void setP(PlayList p) {
        P = p;
    }

    public CD getC() {
        return c;
    }

    public void setC(CD c) {
        this.c = c;
    }

    public CDPlayer(CD c){
        this.c =c;
    }

    public CDPlayer(PlayList p){
        this.P = p;
    }

    public void playNextSong(){
        P.playNextSong();
    }
}
