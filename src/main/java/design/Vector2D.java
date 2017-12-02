package design;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

/**
 * Created by tao on 9/1/17.
 */
public class Vector2D implements Iterator<Integer> {

    private Queue<Iterator<Integer>>q=null;
    public Vector2D(List<List<Integer>>vec2d){
        q=new LinkedList<>();
        for(List<Integer>vec:vec2d){
            q.offer(vec.iterator());
        }
    }
    public Integer next(){
        return q.peek().next();
    }

    public boolean hasNext(){
        while(!q.isEmpty() && !q.peek().hasNext()){
            q.poll();
        }
        return !q.isEmpty() && q.peek().hasNext();
    }
}
