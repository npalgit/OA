package design.jukebox;

import java.util.Set;

/**
 * Created by tao on 10/24/17.
 */
public class JukeBox {
    private CDPlayer cdPlayer;
    private User user;
    private Set<CD>cdCollection;
    private PlayList playList;
    private boolean activated;

    public JukeBox(CDPlayer cdPlayer, User user,Set<CD>cdCollection,PlayList playList){
        this.cdCollection =cdCollection;
        this.cdPlayer = cdPlayer;
        this.user = user;
        this.playList = playList;
        activated = false;
    }

    public void palyNextSong(){
        if(activated == false){
            System.out.println("please insert coin");
        }else{
            playList.playNextSong();
            activated = false;
        }
    }

    public boolean insertCoin(){
        if(activated == false){
            activated = true;
        }
        return true;
    }

}
