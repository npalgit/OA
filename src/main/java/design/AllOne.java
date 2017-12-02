package design;

import java.util.*;

/**
 * Created by tao on 9/21/17.
 */
public class AllOne {

    Map<String,Integer> map=null;
    TreeMap<Integer,Set<String>>cnt=null;
    /** Initialize your data structure here. */
    public AllOne() {
        map=new HashMap<>();
        cnt = new TreeMap<>();
    }

    /** Inserts a new key <Key> with value 1. Or increments an existing key by 1. */
    public void inc(String key) {
        int val = map.getOrDefault(key,0);
        map.put(key,val+1);
        if(!cnt.containsKey(val+1))
            cnt.put(val+1,new HashSet<>());
        cnt.get(val+1).add(key);
        if(cnt.containsKey(val)){
            cnt.get(val).remove(key);
            if(cnt.get(val).isEmpty())
                cnt.remove(val);
        }
    }

    /** Decrements an existing key by 1. If Key's value is 1, remove it from the data structure. */
    public void dec(String key) {
        if(map.containsKey(key)){
            int val = map.get(key);
            if(val==1){
                map.remove(key);
                cnt.get(1).remove(key);
                if(cnt.get(1).isEmpty())
                    cnt.remove(1);
            }
            else{
                map.put(key,val-1);
                cnt.get(val).remove(key);
                if(cnt.get(val).isEmpty())
                    cnt.remove(val);
                if(cnt.get(val-1)==null)
                    cnt.put(val-1,new HashSet<>());
                cnt.get(val-1).add(key);
            }

        }
    }

    /** Returns one of the keys with maximal value. */
    public String getMaxKey() {
        if(map.isEmpty())
            return "";
        return cnt.lastEntry().getValue().iterator().next();
    }

    /** Returns one of the keys with Minimal value. */
    public String getMinKey() {
        if(map.isEmpty())
            return "";
        return cnt.firstEntry().getValue().iterator().next();
    }
}
