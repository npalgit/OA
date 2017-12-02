package design.chat;

import java.util.Date;

/**
 * Created by tao on 10/24/17.
 */
public class AddRequest {
    private  User fromUser;
    private User toUser;
    private Date date;
    RequestStatus status;

    public AddRequest(User from, User to, Date date){
        fromUser = from;
        toUser = to;
        this.date = date;
        status = RequestStatus.Unread;
    }

    public RequestStatus getStatus(){
         return status;
    }

    public User getFromUser(){
        return fromUser;
    }

    public User getToUser(){
        return toUser;
    }

    public Date getDate(){
        return date;
    }
}
