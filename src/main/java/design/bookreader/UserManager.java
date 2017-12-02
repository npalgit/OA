package design.bookreader;

import java.util.Map;

/**
 * Created by tao on 10/24/17.
 */
public class UserManager {

    private Map<Integer,User> users;
    public User addUser(int id, String details, int accountType){
        if(users.containsKey(id))
            return null;
        User user = new User(id,details,accountType);
        users.put(id,user);
        return user;
    }

    public boolean remove(User u){
        return remove(u.getUserId());
    }

    public boolean remove(int id){
        if(!users.containsKey(id))
            return false;
        users.remove(id);
        return true;
    }

    public User find(int id){
        return users.get(id);
    }
}
