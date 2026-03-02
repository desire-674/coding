import java.util.*;
public class fibanocci {
    public static void main(String[] args) {
        Scanner sc= new Scanner(System.in);
System.out.print("Enter a number: ");
int n = sc.nextInt();
int a=0,b=1;
System.out.print("Fibanocci series: ");
for(int i=0;i<n;i++)
{
    System.out.print(a + " ");
    int c=a+b;
    a=b;
    b=c;
}

    }
}