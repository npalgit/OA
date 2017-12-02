package design;

import java.util.*;

/**
 * Created by tao o
 *
 * n 9/14/17.
 */
public class RandomizedCollection {


    /** Initialize your data structure here. */
    Map<Integer,List<Integer>> map=null;
    List<Integer>nums = null;
    Random random = null;
    public RandomizedCollection() {
        nums = new ArrayList<>();
        map = new HashMap<>();
        random = new Random();
    }

    /** Inserts a value to the collection. Returns true if the collection did not already contain the specified element. */
    public boolean insert(int val) {
        boolean hasVal = map.containsKey(val);
        if(!hasVal)
            map.put(val,new ArrayList<>());
        map.get(val).add(nums.size());
        nums.add(val);
        return !hasVal;
    }

    /** Removes a value from the collection. Returns true if the collection contained the specified element. */
    public boolean remove(int val) {
        if(!map.containsKey(val))
            return false;
        List<Integer>indexs = map.get(val);
        int size = indexs.size();
        int ind = indexs.get(size-1);
        indexs.remove(size-1);
        if(size==1)
            map.remove(val);
        int n = nums.size();
        if(ind!=n-1){
            nums.set(ind,nums.get(n-1));
            int nn = map.get(nums.get(n-1)).size();
            map.get(nums.get(n-1)).remove(nn-1);
            map.get(nums.get(n-1)).add(ind);
        }else{

        }
        nums.remove(n-1);
        return true;
    }

    /** Get a random element from the collection. */
    public int getRandom() {
        int size = nums.size();
        return nums.get(random.nextInt(size));
    }
}
