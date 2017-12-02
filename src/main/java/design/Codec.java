package design;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by tao on 9/3/17.
 */
public class Codec {
    public String encode(List<String> strs) {
        StringBuilder sb = new StringBuilder();
        for(String str:strs){
            sb.append(str.length()).append('@').append(str);
        }
        return sb.toString();
    }

    // Decodes a single string to a list of strings.
    public List<String> decode(String s) {
        List<String> res=new ArrayList<>();
        int n = s.length(),i=0;
        int start=0;
        while(i<n){
            start=i;
            while(i<n && Character.isDigit(s.charAt(i))&&s.charAt(i)!='@'){
                i++;
            }
            int len = Integer.parseInt(s.substring(start,i));
            res.add(s.substring(i+1,i+1+len));
            i+=len+1;
        }
        return res;
    }
}
