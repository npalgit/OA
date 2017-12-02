package design.chat;

import java.util.Date;

/**
 * Created by tao on 10/24/17.
 */
public class Message {

    private String content;
    private Date date;
    public Message(String content, Date date){
        this.content = content;
        this.date = date;
    }

    public String getContent(){
        return content;
    }

    public Date getDate(){
        return date;
    }
}
