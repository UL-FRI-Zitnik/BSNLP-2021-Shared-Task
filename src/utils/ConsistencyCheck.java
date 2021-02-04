package src.utils;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.StringTokenizer;

/*
 * Checks consistency of the annotation files (including format, valid entity types, id assignment, etc.) 
 * in a given directory and outputs a report to a file
 * 
 * arg1 - path to the directory with annotation files
 * arg2 - path to the file where the report will be written to 
 */

public class ConsistencyCheck {
  	
  private static List<String> getFilesOfDirectory(String path) 
   { List<String> filePaths = new ArrayList<String>();	 
	 File directory = new File(path);
	 if (!directory.isDirectory()) 
	  { return null;
	  }
	 File[] children = directory.listFiles();
	 for (File child:children) 
	   if(child.isFile()) 
	     filePaths.add(child.getAbsolutePath());			 
	 return filePaths;
   }
	
  public static void main(String[] args) throws UnsupportedEncodingException, FileNotFoundException
	{ String dirPath = args[0];	   
	  String outputFile = args[1];
	  System.setOut(new PrintStream(new FileOutputStream(outputFile), true, "UTF-8"));
	  List<String> filePaths = getFilesOfDirectory(dirPath); 
	  if(filePaths==null)
	   { System.out.println(dirPath + ": is not a directory");
		 System.exit(0); 
	   }		 
	  int numAnnotations = 0;
	  int numFiles = 0;
	  int ORG = 0;
	  int PER = 0;
	  int LOC = 0;
	  int PRO = 0;
	  int EVT = 0;
	  int DIFF_BASE = 0;
	  HashMap<String,String> annotations = new HashMap<String,String>();  
      for(String filePath : filePaths)	  
	   { try { System.out.println("CHECKING FILE: " + filePath);
		       BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(filePath), "UTF-8"));	   
	           String docID = in.readLine();
	           StringTokenizer st = new StringTokenizer(docID,"\t ");
	           if(st.countTokens()!=1)
	            { System.out.println("ERROR: INCORRECT ID: " + docID);   
	            }   
	           int count = 0;
	           numFiles++;
	           while(in.ready())
	            { String entry = in.readLine();
	              st = new StringTokenizer(entry,"\t");
	              count++;
	              numAnnotations++;
	              if(st.countTokens()==0)
	               { System.out.println("WARRNING: LINE: " + count + " is empty");
	                 continue;	            	  
	               }
	              if(st.countTokens()!=4)
	               { System.out.println("ERROR: WRONG NUMBER OF ITEMS (" + st.countTokens() + ") IN LINE: " + count + " :" + entry);
	                 continue;
	               }
	              String form = st.nextToken();
	              String base = st.nextToken();
	              String type = st.nextToken();
	              String id = st.nextToken();
	              String code = annotations.get(form);
	              if(code==null)
	            	 annotations.put(form, id); 
	              else
	                { if(code.compareTo(id)!=0)
	            	    System.out.println("WARNING: " + form + " at line: " + count + " has an ID: \"" + id + "\" that is different from the ID \"" + code + "\" assigned to the same form elsewhere");
	                }
	              if(form.compareTo(base)!=0)
	            	  { DIFF_BASE++;                        	            	 
	            	  }
	              if((type.compareTo("PER")!=0)&&(type.compareTo("ORG")!=0)&&(type.compareTo("LOC")!=0)&&(type.compareTo("PRO")!=0)&&(type.compareTo("EVT")!=0))
	               { System.out.println("ERROR: UNKNOWN TYPE " + type + " IN LINE: " + count + " :" + entry);
	                 continue;
	               }	 
	              if(type.compareTo("PER")==0)
	            	 PER++;
	              else if(type.compareTo("ORG")==0)
	                     ORG++;
	              else if(type.compareTo("LOC")==0)
	                     LOC++;
	              else if(type.compareTo("PRO")==0)
	                     PRO++;
	              else if(type.compareTo("EVT")==0)
	                     EVT++;	              
	           }
	          in.close();	     
	        }
	     catch(Exception e)
	      { System.out.println("ERROR WHILE PROCESSING FILE: " + filePath);		   
	      }	  
	    System.out.println("---------------------------------------------------------");
	   
	 }
    System.out.println("Total number of annotations: " + numAnnotations);
    System.out.println("ORG: " + ORG);
    System.out.println("LOC: " + LOC);
    System.out.println("PER: " + PER);
    System.out.println("PRO: " + PRO);    
    System.out.println("EVT: " + EVT);
    System.out.println("MENTIONS THAT ARE INFLECTED VARIANTS: " + DIFF_BASE);
    System.out.println("Total number of files: " + numFiles);  
  } 
}  
