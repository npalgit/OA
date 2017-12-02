package design.chat;

import java.util.ArrayList;

/**
 * Created by tao on 10/24/17.
 */
public class Conversation {
    protected ArrayList<User> participants = new ArrayList<>();
    protected int id;
    protected ArrayList<Message>messages = new ArrayList<>();

    public ArrayList<Message> getMessages(){
        return messages;
    }

    public boolean addMessage(Message m){
        messages.add(m);
        return true;
    }

    public int getId(){
        return id;
    }
}
