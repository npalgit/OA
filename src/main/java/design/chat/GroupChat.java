package design.chat;

/**
 * Created by tao on 10/24/17.
 */
public class GroupChat extends Conversation {
    public void removeParticipant(User user){
        participants.remove(user);
    }

    public void addParticipant(User user){
        participants.add(user);
    }
}
