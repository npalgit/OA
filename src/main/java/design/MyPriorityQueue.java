package design;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by tao on 10/18/17.
 */
public class MyPriorityQueue {

    List<Integer>nums = null;
    public MyPriorityQueue(){
        nums=new ArrayList<>();
        nums.add(0);
    }

    public int getLeft(int ind){
        return ind*2;
    }

    public int getRight(int ind){
        return ind*2+1;
    }

    public int getParent(int ind){
        return ind/2;
    }

    public void heapify(int ind){
        int size = nums.size();
        if(size<=2)
            return;
        int l = getLeft(ind);
        int r = getRight(ind);
        int minInd = ind;
        if(l<size && nums.get(l)<nums.get(minInd))
            minInd = l;
        if(r<size && nums.get(r)<nums.get(minInd))
            minInd = r;
        if(minInd!=ind){
            int tmp = nums.get(minInd);
            nums.set(minInd,nums.get(ind));
            nums.set(ind,tmp);
            heapify(minInd);
        }
    }


    public void buildHeap(){
        int n = (nums.size()-1)/2;
        for(int i=n;i>=1;--i)
            heapify(i);
    }

    public int peek(){
        if(isEmpty())
            throw new IndexOutOfBoundsException();
        buildHeap();
        return nums.get(1);
    }

    public void offer(int ele){
        nums.add(ele);
    }

    public int pop(){
        if(isEmpty())
            throw new IndexOutOfBoundsException();
        buildHeap();
        int val = nums.get(1);
        int size = nums.size();
        nums.set(1,nums.get(size-1));
        nums.remove(size-1);
        return val;
    }

    public boolean isEmpty(){
        return nums.size()<=1;
    }

    public boolean remove(int ele){
        int n = nums.size();
        int i=1;
        for(;i<n;++i){
            if(nums.get(i)==ele)
                break;
        }
        if(i==n)
            return false;
        nums.set(i,nums.get(n-1));
        nums.remove(n-1);
        return true;
    }
}
