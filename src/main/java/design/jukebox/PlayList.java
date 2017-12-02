package design.jukebox;

import java.util.Queue;

/**
 * Created by tao on 10/24/17.
 */
public class PlayList {

    private Song currentSong;
    private Queue<Song> queue;

    public PlayList(Queue<Song>q){
        this.queue = q;
    }

    public void playNextSong(){
        currentSong = queue.poll();
    }

    public Song getNextSong(){
        return queue.peek();
    }

    public void addSong(Song s){
        queue.add(s);
    }
}
