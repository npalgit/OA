package design;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by tao on 10/18/17.
 */
public class MyPriorityQueueTest {

    @Test
    public void TestMyPriority(){
        MyPriorityQueue pq = new MyPriorityQueue();
        pq.offer(10);
        pq.offer(-3);
        pq.offer(50);
        pq.offer(4);
        System.out.println(pq.pop());
        System.out.println("------");
        pq.offer(4);
        pq.offer(-100);
        pq.offer(-2);
        pq.remove(4);
        pq.remove(-101);
        while(!pq.isEmpty()){
            System.out.println(pq.pop());
        }

    }

}