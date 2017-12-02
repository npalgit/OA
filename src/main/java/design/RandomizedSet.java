package design;

import java.util.*;

/**
 * Created by tao on 9/13/17.
 */
public class RandomizedSet {
    public List<Integer>num=null;
    Random random =null;
    public Map<Integer,Integer>map=null;
    /** Initialize your data structure here. */
    public RandomizedSet() {
        num=new ArrayList<>();
        map=new HashMap<>();
        random=new Random();
    }

    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
        if(map.containsKey(val))
            return false;
        map.put(val,num.size());
        num.add(val);
        return true;
    }

    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
        if(!map.containsKey(val))
            return false;
        //change with last index
        int size = num.size();
        if(map.get(val)!=size-1){
            int tail = num.get(size-1);
            int ind = map.get(val);
            map.put(tail,ind);
            num.set(ind,tail);
        }
        map.remove(val);
        num.remove(size-1);
        return true;
    }

    /** Get a random element from the set. */
    public int getRandom() {
        int size = num.size();
        return num.get(random.nextInt(size));
    }
}
